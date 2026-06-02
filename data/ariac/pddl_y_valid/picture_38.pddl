(define (problem picture_38)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        green_regulator red_regulator red_battery - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear green_regulator)
        (clear red_regulator)
        (clear red_battery)
        (part_at green_regulator table)
        (part_at red_regulator table)
        (part_at red_battery table)
    )
    
    (:goal
        (and
            (part_at green_regulator regulator_placement)
            (part_at red_battery battery_placement)
        )
    )
)
