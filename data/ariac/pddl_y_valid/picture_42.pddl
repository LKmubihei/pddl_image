(define (problem picture_42)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        red_regulator green_regulator red_battery - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear red_battery)
        (clear green_regulator)
        (clear red_regulator)
        (part_at red_regulator table)
        (part_at green_regulator table)
        (part_at red_battery table)
    )
    
    (:goal
        (and
            (part_at red_regulator regulator_placement)
            (part_at red_battery battery_placement)
        )
    )
)
