(define (problem picture_48)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        green_battery red_sensor red_battery - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear green_battery)
        (clear red_sensor)
        (clear red_battery)
        (part_at green_battery table)
        (part_at red_sensor table)
        (part_at red_battery table)
    )
    
    (:goal
        (and
            (part_at green_battery battery_placement)
        )
    )
)
