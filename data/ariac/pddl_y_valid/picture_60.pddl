(define (problem picture_60)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        red_battery blue_pump red_regulator - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear red_battery)
        (clear red_regulator)
        (on red_battery blue_pump)
        (part_at blue_pump table)
        (part_at red_regulator table)
    )
    
    (:goal
        (and
            (part_at blue_pump pump_placement)
            (part_at red_regulator regulator_placement)
            (part_at red_battery battery_placement)
        )
    )
)
